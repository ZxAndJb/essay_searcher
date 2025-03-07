from langchain_openai import ChatOpenAI
from utils.prompts import restate_system_message, judge_system_message, response_system_message
from utils.data_utils import parse_dict_to_string, parse_search_dict_to_string
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
from database import essay_database
from typing import TypedDict, Annotated, List
from langgraph.graph import START, END, StateGraph
from langgraph.constants import Send
import operator
import asyncio
import os



class AgentState(TypedDict):
    user_query: str
    restate_query: str
    document: List[dict]
    cur_check: dict
    filtered_document: Annotated[list[dict], operator.add]

class Essay_Agent():
    def __init__(self, model_name: str, url: str, knowledge_base: essay_database):
        self.llm = ChatOpenAI(base_url=url, model=model_name, temperature=0.6)
        self.parser = StrOutputParser()
        self.knowledge_base = knowledge_base
        self.default_collection = self.knowledge_base.client.list_collections()[0]
        self.search_tools = TavilySearchResults(max_results=5, search_depth="advanced")

        self.graph = StateGraph(AgentState)
        self.graph.add_node("restate", self.restate_query)
        self.graph.add_node("search", self.rag_search)
        self.graph.add_node("judge", self.judge_relevance)
        self.graph.add_node("summary", self.summarize)
        self.graph.add_node("search_online", self.searchonline)

        self.graph.add_conditional_edges("search", self.filter_results)
        self.graph.add_conditional_edges("judge", self.is_summarize, {True: "search_online", False: "summary"})

        self.graph.add_edge(START, "restate")
        self.graph.add_edge("restate", "search")
        self.graph.add_edge("summary", END)
        self.graph.add_edge("search_online", END)

        self.graph = self.graph.compile()

    def restate_query(self, state: AgentState):
        restate_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", '{restate_system_message}'),
                ('human', '{query}')
            ],
        )

        ask_chain = restate_prompt | self.llm | self.parser
        llm_restate = ask_chain.invoke({'restate_system_message': restate_system_message, "query":state['user_query'] })
        return {"raw_query": state['user_query'], "restate_query": llm_restate}

    def rag_search(self, state: AgentState):
        print(f"RAG searching for query: {state['restate_query']}")
        res = self.knowledge_base.search(state['restate_query'], self.default_collection)
        print("Finish searching")
        return {"raw_query":state['user_query'], 'document': res}

    def searchonline(self, state: AgentState):
        print("未检测到您提供的知识库中有相关问题的内容，我们已为您联网查询相关信息，请稍等！")
        res = self.search_tools.invoke(state['restate_query'])
        print(parse_search_dict_to_string(res))


    # Send, Conditional edge
    def filter_results(self, state: AgentState):
        return [Send("judge", {"cur_check": doc, "restate_query": state['restate_query']}) for doc in state['document']]

    # 数据处理逻辑
    def judge_relevance(self, state: AgentState):
        content = state['cur_check']['content']
        judge_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", judge_system_message),
                ('human', '{query}')
            ],
        )
        ask_chain = judge_prompt | self.llm | self.parser
        res = ask_chain.invoke({"content": content, "query": state['restate_query']})
        if "不相关" in res:
            return {}
        else:
            return {'filtered_document': [state['cur_check']]}

    def is_summarize(self, state: AgentState):
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
        for resp in ask_chain.stream({"content": content, "query": state['restate_query']}):
            print(resp,  end="")

        return {}

    def __call__(self):
        print("您好，我是您的论文查询助手，请输入您的查询！")
        # query = input()
        query ='我想要知道deepseek MLA是如何进行计算的'
        self.graph.invoke({"user_query": query})


if __name__ == '__main__':
    uri='http://localhost:19530'
    db_name='essay_seacher_pdfs'
    model_path = "/mnt/d/PycharmCode/LLMscratch/essay_searcher/embedding_models/BAAI/bge-m3"
    database = essay_database(uri, db_name, model_path)
    agent = Essay_Agent(url="https://dashscope.aliyuncs.com/compatible-mode/v1", model_name="qwen-plus", knowledge_base=database)
    agent()

    graph_png = agent.graph.get_graph().draw_mermaid_png()
    with open("./agent_workflow.png","wb") as f:
        f.write(graph_png)