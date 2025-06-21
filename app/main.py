# from llm.vector import retriever
from llm.brain import Brain, LLM_MODEL

brain = Brain(LLM_MODEL.DEEPSEEK_R1_8B)

while True:
    print("\n\n-------------------------------")
    question = input("Ask your question (q to quit): ")
    print("\n\n")
    if question == "q":
        break;

    # context = retriever.invoke(question)
    result = brain.chat("18192", "", question)
    print(result)