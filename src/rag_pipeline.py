

from retriever import load_retriever, retrieve, format_context
from generator import load_generator, build_prompt, generate_answer
from typing import Optional


class RAGPipeline:
   

    def __init__(
        self,
        vector_store_dir: str = '../vector_store',
        generator_model:  str = 'google/flan-t5-large',
        k: int = 5,
        max_new_tokens: int = 300,
        device: int = -1,
    ):
        
        print("Initialising RAG pipeline...")

        self.k = k

        self.index, self.chunks_df, self.embed_model = load_retriever(
            vector_store_dir=vector_store_dir
        )

        self.generator = load_generator(
            model_name=generator_model,
            max_new_tokens=max_new_tokens,
            device=device,
        )

        print("RAG pipeline ready.\n")

    def ask(
        self,
        question: str,
        product_filter: Optional[str] = None,
    ) -> dict:
       
        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        print(f"Question: {question}")
        if product_filter:
            print(f"Filter  : {product_filter}")

        sources = retrieve(
            query=question,
            index=self.index,
            chunks_df=self.chunks_df,
            model=self.embed_model,
            k=self.k,
            product_filter=product_filter,
        )

        context = format_context(sources)

        prompt = build_prompt(question=question, context=context)

        answer = generate_answer(prompt=prompt, generator=self.generator)

        print(f"Answer  : {answer[:150]}...")

        return {
            'question': question,
            'answer':   answer,
            'sources':  sources,
            'prompt':   prompt,
        }