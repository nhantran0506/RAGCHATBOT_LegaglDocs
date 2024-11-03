from processing import RAG
import logging
logger = logging.getLogger(__name__)

rag = RAG(model_name="gpt-4o-mini")

while True:
    try:
        user_input = input("Người dùng: ")
        if user_input.strip() == "END":
            break
        
        response = rag.generate_response(user_input)
        print("CHATBOT: ", response)
        print()
    except KeyboardInterrupt:
        break
    

    
