# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"

# from typing import Any, Text, Dict, List
#
# from rasa_sdk import Action, Tracker
# from rasa_sdk.executor import CollectingDispatcher
#
#
# class ActionHelloWorld(Action):
#
#     def name(self) -> Text:
#         return "action_hello_world"
#
#     def run(self, dispatcher: CollectingDispatcher,
#             tracker: Tracker,
#             domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
#
#         dispatcher.utter_message(text="Hello World!")
#
#         return []
"""
import requests 
from typing import Any, Text, Dict, List 
from rasa_sdk import Action, Tracker 
from rasa_sdk.executor import CollectingDispatcher 
class ActionLlama3Explain(Action):  
  def name(self) -> Text:  
    return "action_llama3_explain" 
  def run(self, dispatcher: CollectingDispatcher,  
    tracker: Tracker,  
    domain: Dict[Text, Any]) -> List[Dict[Text, Any]]: 

    user_message = tracker.latest_message.get("text") 
 
    if not user_message: 
      dispatcher.utter_message(text="Can you please repeat your question?") 
      return [] 
    try: 
      response = requests.post( 
        "http://localhost:11434/api/generate", 
        json={ 
          "model": "deepseek-r1:1.5b", 
          "prompt": user_message, 
          "stream": False 
        } 
      ) 

      if response.status_code == 200: 
        data = response.json() 
        answer = data.get("response", "").strip() 
        if answer: 
          dispatcher.utter_message(text=answer) 
        else: 
          dispatcher.utter_message(text="I couldn't find an explanation for that.") 
      else: 
        print(f"Ollama API Error - Status Code: {response.status_code}") 
        dispatcher.utter_message(text="Sorry, I had trouble generating the answer.") 

    except Exception as e: 
      print(f"Error in Ollama request: {e}") 
      dispatcher.utter_message(text="Sorry, something went wrong while contacting the model.") 
"""
"""
from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from .rag_system import rag_system

class ActionRAGLookup(Action):
    def name(self) -> Text:
        return "action_rag_lookup"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        try:
            # Vérifier si le système RAG est chargé
            if rag_system.index is None:
                rag_system.load_knowledge_base()
            
            # Récupérer la requête utilisateur
            user_query = tracker.latest_message.get('text')
            
            if not user_query:
                dispatcher.utter_message(text="Je n'ai pas compris votre question. Pouvez-vous reformuler ?")
                return []
            
            # Rechercher dans la base de connaissances
            results = rag_system.search(user_query, top_k=1)
            
            if results:
                best_answer = results[0]['answer']
                dispatcher.utter_message(text=best_answer)
            else:
                dispatcher.utter_message(text="Je n'ai pas trouvé d'information pertinente dans ma base de connaissances.")
            
        except Exception as e:
            print(f"Erreur RAG: {e}")
            dispatcher.utter_message(text="Désolé, j'ai rencontré un problème technique. Veuillez réessayer.")
        
        return []
"""





from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
# On importe tes variables et fonctions depuis ton fichier indexation.py
from indexation import client, model, collection_name, chercher_faq
import re

#implentation du request routing pour decter si la question du user en russe

#def is_russian(text):
    # Utilise une expression régulière pour détecter les caractères cyrilliques
   # return bool(re.search('[а-яА-Я]', text))

# Dans ton action Rasa :

#if is_russian(user_query):
    # Ici, on appelle Ollama uniquement pour traduire du russe au français
 #   query_for_rag = call_ollama_to_translate(user_query)
#else:
    # On utilise la question telle quelle pour le français ou l'anglais
#    query_for_rag = user_query

# fin de l'implementation

class ActionSearchFAQ(Action):
    def name(self) -> Text:
        return "action_search_faq"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        # 1. On récupère le message de l'utilisateur
        user_message = tracker.latest_message.get('text')

        # 2. On cherche dans Qdrant
        reponses = chercher_faq(client, collection_name, model, user_message, limite=1)

        # 3. On traite le résultat si le score est bon
        if reponses and reponses[0].score > 0.50:
            p = reponses[0].payload
            
            # --- Extraction des données ---
            # On prend 'content' s'il existe, sinon 'texte_fr'
            fr = p.get('content') or p.get('texte_fr') or "Texte non trouvé"
            ru = p.get('texte_ru', "Version russe non disponible")
            src = p.get('source', "#") # '#' est un lien vide par défaut
            pg = p.get('page', "?")

            # --- Construction du message final ---
            # Le \n crée un retour à la ligne
            message = (
                f"🇫🇷 **Français :**\n{fr}\n\n"
                f"🇷🇺 **Русский :**\n{ru}\n\n"
                f"📂 **Source :** [Consulter le document (Page {pg})]({src})"
            )

            # 4. Envoi au bot
            dispatcher.utter_message(text=message)
            print(f"✅ Réponse envoyée (Score: {reponses[0].score:.4f})")
        
        else:
            dispatcher.utter_message(text="Désolé, je n'ai pas trouvé d'information précise.")
            print("⚠️ Aucun résultat ou score trop bas.")

        return []