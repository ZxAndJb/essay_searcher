from langchain_openai import ChatOpenAI
from utils.prompts import restate_system_message, judge_system_message, response_system_message, answering_system_messgae
from utils.data_utils import parse_dict_to_string, parse_search_dict_to_string
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
from database import essay_database
from typing import TypedDict, Annotated, List
from langgraph.graph import START, END, StateGraph
from langgraph.constants import Send
from langgraph.prebuilt import ToolNode
import operator
from langchain_core.messages import HumanMessage, SystemMessage, RemoveMessage, BaseMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
import os
from langgraph.prebuilt import tools_condition



os.environ['LANGCHAIN_TRACING_V2'] =
os.environ['LANGCHAIN_API_KEY'] =
os.environ["OPENAI_API_BASE"] =
# os.environ['OPENAI_API_KEY'] =
os.environ['OPENAI_API_KEY'] =
os.environ['TAVILY_API_KEY'] =
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    context_summary: str
    raw_query: str
    restate_query: str
    filtered_document: Annotated[list[dict], operator.add]

class checkState(TypedDict):
    query: str
    document : dict


class Essay_Agent():
    def __init__(self, model_name: str, knowledge_base: essay_database):
        self.tool = [self.rag_search]
        self.llm = ChatOpenAI(model=model_name, temperature=0.6).bind_tools(self.tool)
        self.parser = StrOutputParser()
        self.knowledge_base = knowledge_base
        self.default_collection = self.knowledge_base.client.list_collections()[0]
        self.search_tools = TavilySearchResults(max_results=5, search_depth="advanced")
        self.memory = MemorySaver()

        self.graph = StateGraph(AgentState)
        self.graph.add_node("summary_context", self.summary_context)
        self.graph.add_node("restate", self.restate_query)
        self.graph.add_node('answer', self.answer)
        self.graph.add_node("tools", ToolNode([self.rag_search]))
        self.graph.add_node("judge", self.judge_relevance)
        self.graph.add_node("summary", self.summarize)
        self.graph.add_node("search_online", self.searchonline)
        self.graph.add_node("reduce_judge", self.reduce_judge)

        self.graph.add_conditional_edges("tools", self.filter_results, ['judge'])
        self.graph.add_conditional_edges("reduce_judge", self.if_summarize, {True: "search_online", False: "summary"})
        self.graph.add_conditional_edges("answer", self.if_go_rag, [END, 'tools'])

        self.graph.add_edge(START, "summary_context")
        self.graph.add_edge("summary_context", "restate")
        self.graph.add_edge("restate", "answer")
        self.graph.add_edge('tools', 'judge')
        self.graph.add_edge("judge", "reduce_judge")
        self.graph.add_edge("summary", END)
        self.graph.add_edge("search_online", END)

        self.graph = self.graph.compile(checkpointer=self.memory)


    def summary_context(self, state: AgentState):
        if len(state['messages'])>=6:
            if state.get('context_summary', None) is not None:
                summary_context_message = f"这是关于之前对话的总结：{state['context_summary']}\n， 请你根据以下最新得对话更新上面的总结。"
            else:
                summary_context_message = "请你根据根据上面的对话内容生成一个总结："

            messages = state['messages'] + [HumanMessage(summary_context_message)]
            summary_chain =  self.llm | self.parser
            response = summary_chain.invoke(messages)
            delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]

            return {"context_summary": response, 'messages': delete_messages}

        else:
            return state


    def if_go_rag(self, state: AgentState):
        last_message = state['messages'][-1]
        if hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0:
            return "tools"
        else:
            print(last_message.content)
            return "__end__"


    def answer(self, state: AgentState):
        summary = state.get("context_summary", None)
        if summary:
            system_messgaes = f'这是关于之前对话的总结：{summary}，请你在必要时参考它去回答用户的问题'
            messages = [SystemMessage(content=system_messgaes)] + state["messages"] + [HumanMessage(state['restate_query'])]
        else:
            messages = state["messages"] + [HumanMessage(state['restate_query'])]
        response = self.llm.invoke(messages)
        return {"messages": [response]}



    def restate_query(self, state: AgentState) -> AgentState:
        restate_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", '{restate_system_message}'),
                ('human', '{query}')
            ],
        )

        ask_chain = restate_prompt | self.llm | self.parser
        llm_restate = ask_chain.invoke({'restate_system_message': restate_system_message, "query":state['raw_query'] })
        return {"restate_query": llm_restate, 'messages': [HumanMessage(content =llm_restate)] }

    def rag_search(self, query: str):
        """
        search on the local database when you have no knowledge to answer user queries.
        Args:
        a: user query
        :return: a document list containing relevant file chunks
        """
        print(f"RAG searching for query: {query}")
        res = self.knowledge_base.search(query, self.default_collection)
        print("Finish searching")
        return res

    def searchonline(self, state: AgentState):
        print("未检测到您提供的知识库中有相关问题的内容，我们已为您联网查询相关信息，请稍等！\n")
        res = self.search_tools.invoke(state['restate_query'])
        parser_res = parse_search_dict_to_string(res)
        print(parser_res)
        return {"messages": AIMessage(content = "未检测到您提供的知识库中有相关问题的内容，我们已为您联网查询相关信息，请稍等！\n"+ parser_res) }

    # Send, Conditional edge
    def filter_results(self, state: AgentState):
        jp = JsonOutputParser()
        docs = jp.parse(state['messages'][-1].content)
        return [Send("judge", {"query": state['restate_query'], "document": doc}) for doc in docs]

    # 数据处理逻辑
    def judge_relevance(self, state: checkState) -> AgentState:
        if state:
            content = state['document']['content']
            judge_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", judge_system_message),
                    ('human', '{query}')
                ],
            )
            ask_chain = judge_prompt | self.llm | self.parser
            res = ask_chain.invoke({"content": content, "query": state['query']})
            if "不相关" in res:
                return {}
            else:
                return {'filtered_document': [state['document']]}
        else:
            return {}


    def reduce_judge(self, state: AgentState):
        delete_messages = [RemoveMessage(id=m.id) for m in state['messages'][-2:]]
        return {'messages': delete_messages}

    def if_summarize(self, state: AgentState):
        return len(state['filtered_document']) == 0

    def summarize(self, state: AgentState):
        filter_content = state['filtered_document']
        content = parse_dict_to_string(filter_content)
        judge_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", response_system_message),
                ('human', '{query}')
            ],
        )
        ask_chain = judge_prompt | self.llm | self.parser
        res = ""
        for resp in ask_chain.stream({"content": content, "query": state['restate_query']}):
            res+=resp
            print(resp,  end="")

        return {'messages': [AIMessage(content)]}

    def __call__(self):
        print("您好，我是您的论文查询助手, 您有什么和我想聊的吗？")
        thread = {"configurable": {"thread_id": "1"}}
        query = input()
        # query = '我对LexMatcher算法不太熟悉，我想知道LexMatcher算法的原理'
        self.graph.invoke({"raw_query": query}, thread)


if __name__ == '__main__':
    uri='http://localhost:19530'
    db_name='essay_seacher_pdfs'
    model_path = "/mnt/d/PycharmCode/LLMscratch/essay_searcher/embedding_models/BAAI/bge-m3"
    database = essay_database(uri, db_name, model_path)
    agent = Essay_Agent(model_name="gpt-4o", knowledge_base=database)
    agent()

    graph_png = agent.graph.get_graph().draw_mermaid_png()
    with open("./agent_workflow.png","wb") as f:
        f.write(graph_png)