model_id = "mistralai/Mistral-7B-Instruct-v0.2"
import pandas as pd
import torch
import tqdm
from langchain import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms import VLLM
from langchain_core.output_parsers import StrOutputParser
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers import BitsAndBytesConfig
from langchain import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain
from langchain.prompts import ChatPromptTemplate


llm = VLLM(
            model=model_id,
            trust_remote_code=True,  
            max_new_tokens=6000,
            top_k=10,
            top_p=0.95,
            temperature=0.1,
            do_sample=True,
            enforce_eager=True,
            gpu_memory_utilization = 0.3
            )

prompt = ChatPromptTemplate.from_template("""
[INST] I am an objective and thorough and extremely strict assistant dedicated to rigorously evaluating student responses based solely on the provided question and rubric. I will only see what is explicitly provided to me. I shall do everything as it's said. I must not generate new answers or engage in emotional responses. I will be extremely strict in evaluation and extremely strict in providing marks. I will not be influenced in marking by implicitly implying anything from the student answer. I will mark the student answer only based on the given scoring criteria, and nothing else. I will provide marks only as per the criteria.

**QUESTION:** {question} 
---- The question ends here -----

**RUBRIC:** {rubric} 
------ The rubric ends here -----

**STUDENT ANSWER:** {student_answer} 
------The student answer ends here-----

#Only follow these instructions for evaluation. 
**Scoring:**

* Dont implicitly assume any criterion of rubric with the answer.
* Award 0 for the criterion with only implicit mentions.
* If a particular point met a criterion, do not check the same point of the answer with other criteria.
* Do not award marks for implicit points as they are not explicit to be marked on.
* Award 0 for the criterion, for any implicit match of the answer with the criterion.
* Award 0 for the entire answer, if the entire answer is just a rephrased version of the given question, and you are extremely strictly ordered to not continue evaluating on other criteria. Stop checking other criteria.
* Award 0 marks for the criterion, if there is no string match of that criterion in student answer.
* Award 0 marks for the criterion, if the criteria is only implicitly mentioned in the answer.
* Award 1 mark for the criterion, if the answer has a sentence of at least 10 words of explanation ONLY focussing on that phrase AND also exact match. 
* Award 0.5 mark for the criterion, if there is only a partial string match of that criterion in student answer.

**Example:**

1.
Question - Evaluate the security robustness of blockchain networks against potential vulnerabilities such as 51% attacks and double-spending.
Answer - The analysis evaluates the security robustness of blockchain networks, with a focus on vulnerabilities such as 51% attacks and double-spending.

Marks = 0 (Answer is a simple rephrasing of the question)


**Explanation:**

Provide a **detailed and objective explanation** of the:
* Awarded score based on the rubric.
* **Reasoning** for assigning points or penalties for each criterion, referencing specifics from the rubric and student answer.

**Marks:**

Add the marks achieved for the answer for each criterion and provide the total mark in a separate line.

Student mark: Mark achieved from criterion1 + criterion2 + criterion3 + criterion4 + criterion5
[/INST]
""")

question = input("Question: ")
rubric = input("Rubric: ")
answer = input("Answer: ")
chain = LLMChain(prompt=prompt, llm=llm)
query = {"student_answer": answer, "rubric": rubric, "question": question}
print(chain.invoke(query))


# def chat():
#     print("Hello! I am Mistral 7B and I can help with your document?\nIf you want to stop you can enter STOP at any point!")
#     print()
#     print("-------------------------------------------------------------------------------------")
#     question = input("Question: ")
#     rubric = input("Rubric: ")
#     answer = input("Answer: ")
#     while question != "STOP":
#         query = {"student_answer": answer, "rubric": rubric, "question": question}
#         print(chain.invoke(query))
#         torch.cuda.empty_cache()
#         print("\nIs there anything else you would like my help with?")
#         print("-------------------------------------------------------------------------------------")
#         question = input("Question: ")
#         rubric = input("Rubric: ")
#         answer = input("Answer: ")
	


# chat()
