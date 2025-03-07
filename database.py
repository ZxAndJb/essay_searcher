import glob
import numpy as np
from utils.data_utils import load_md, filter_content, result_paser
from pymilvus import MilvusClient, Collection
from typing import List, Optional, TypedDict
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from pymilvus import FieldSchema, CollectionSchema, DataType

class chunk(TypedDict):
    paper_name: str
    section: str
    content: str
    embedding: List[float]

class essay_database:
    def __init__(self, uri, db_names, model_path):
        self.client = MilvusClient(uri=uri, db_names=db_names)

        self.embedding_model = BGEM3EmbeddingFunction(
            model_name=model_path,  # Specify the model name
            device='cuda:0',  # Specify the device to use, e.g., 'cpu' or 'cuda:0'
            use_fp16=True  # Specify whether to use fp16. Set to `False` if `device` is `cpu`.
        )
        self.chunk_size = 512

    def add_batch(self, datalist, collection_name):
        chunk_list = []
        content_queue = []
        for data in datalist:
            paper_name = data['meta_information']['title']
            for t, c in data['content'].items():
                truncated_content = self.truncate_chunk(c)
                content_queue.append(truncated_content)
                chunk_list.append({
                    "paper_name": paper_name,
                    "section": t,
                    "content": truncated_content
                })
        docs_embeddings = self.embedding_model.encode_documents(content_queue)['dense']
        for emb, chun in zip(docs_embeddings, chunk_list):
            chun['embedding'] = emb.astype(np.float32)

        self.client.insert(
            collection_name=collection_name,
            data=chunk_list,
        )

    def truncate_chunk(self, chunk_content):
        split_content = chunk_content.split(" ")
        chunk_length = len(split_content)
        if chunk_length <= self.chunk_size:
            return chunk_content
        truncated_content = split_content.copy()
        while chunk_length > 512:
            last_newline_index = chunk_content.rfind('\n')

            # 如果找不到换行符或者字符串已经不能再被有效截断，则直接截断到512
            if last_newline_index <= 0 or last_newline_index >= 512:
                return " ".join(split_content[:512])

            truncated_content = truncated_content[:last_newline_index]

        return truncated_content

    def search(self, query, collection_name, max_results=3, retain_fileds = ['paper_name', 'section', 'content']):
        if self.client.get_load_state(collection_name="pdf_chunks")['state'] != 3:
            self.load_collection(collection_name)
        query_vector = self.embedding_model.encode_queries([query])['dense']
        query_vector = [arr.astype(np.float32) for arr in query_vector]
        res = self.client.search(
            collection_name=collection_name,
            anns_field="embedding",
            data=query_vector,
            limit=max_results,
            search_params={"metric_type": "IP"},
            output_fields=['paper_name', 'section', 'content']
        )
        res = result_paser(res, retain_fileds)
        return res
    def load_collection(self, collection_name):
        self.client.load_collection(collection_name=collection_name, replica_number=1)

    def delete(self):
        pass

    def create_collection(self, field_dict: dict, collection_name: str, description: Optional[str], dimension=1024):
        if self.client.has_collection(collection_name=collection_name):
            print(f"The collection {collection_name} has been existed")
            return
        fields_schema = [FieldSchema(name='id', dtype=DataType.INT64, is_primary=True, auto_id=True)]
        for k, v in field_dict.items():
            if v == 'DataType.FLOAT_VECTOR':
                field = FieldSchema(name=k, dtype=DataType.FLOAT_VECTOR, dim=dimension)
            elif v == 'DataType.INT64':
                field = FieldSchema(name=k, dtype=DataType.INT64)
            elif v == 'STRING':
                field = FieldSchema(name=k, dtype=DataType.VARCHAR, max_length=10240)
            else:
                continue
            fields_schema.append(field)

        schema = CollectionSchema(fields=fields_schema)
        self.client.create_collection(
            collection_name=collection_name,
            schema=schema,  # 使用刚才创建的schema
            dimension=dimension,  # The vectors we will use in this demo has 768 dimensions,

            descrption=description
        )
        collection = Collection(name=collection_name,using=self.client._using)

        index_params = {
            "index_type": "FLAT",
            "metric_type": "IP",
        }
        collection.create_index(field_name="embedding", index_params=index_params)


if __name__ == '__main__':
    uri='http://localhost:19530'
    db_name='essay_seacher_pdfs'
    model_path = "/mnt/d/PycharmCode/LLMscratch/essay_searcher/embedding_models/BAAI/bge-m3"

    # md_folder_path = "/mnt/d/PycharmCode/LLMscratch/essay_searcher/raw_data/mds"
    # md_files = glob.glob(f'{md_folder_path}/*.md')
    # data_list = []
    # for f in md_files:
    #     data_list.append(load_md(f))
    database = essay_database(uri, db_name, model_path)
    query = "我记得有个模型超越了ALMA，请你告诉我他的名字"
    database.search(query,'pdf_chunks')
    # field_dict = {
    #     'paper_name': 'STRING',
    #     'section': 'STRING',
    #     'content': 'STRING',
    #     'embedding': 'DataType.FLOAT_VECTOR'
    # }
    #
    # database.create_collection(
    #     field_dict=field_dict,
    #     collection_name='pdf_chunks',
    #     description="Knowledge base",
    #     dimension=1024  # The vectors we will use in this demo has 384 dimensions
    # )
    # database.add_batch(data_list, 'pdf_chunks')
