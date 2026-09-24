import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENAI_API_KEY"),
)

SYSTEM_PROMPT = """Du er en matematikklærer for ingeniørstudenter. Svar KUN i gyldig JSON-format.
Formatet må være nøyaktig slik:
{
  "svar": "sluttresultat her (f.eks. 4)",
  "steg": ["steg 1 forklaring", "steg 2 forklaring"],
  "formler_brukt": []
}"""

def solve_task(oppgave: str) -> dict:
    try:
        response = client.chat.completions.create(
            model="nvidia/nemotron-3-ultra-550b-a55b:free",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": oppgave}
            ],
            temperature=0.2
        )

        content = response.choices[0].message.content or "{}"
        
        # Vask bort eventuell markdown-formatering hvis modellen legger til ```json ... ```
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        data = json.loads(content)
        
        tokens_used = response.usage.total_tokens if (response and response.usage) else 0

        return {
            "svar": str(data.get("svar", "Ingen svar levert")),
            "steg": data.get("steg", []),
            "formler_brukt": data.get("formler_brukt", []),
            "verifisert": False,
            "tokens_brukt": tokens_used,
            "estimert_kostnad": 0.0
        }

    except Exception as e:
        return {
            "svar": f"Feil ved henting av svar: {str(e)}",
            "steg": [],
            "formler_brukt": [],
            "verifisert": False,
            "tokens_brukt": 0,
            "estimert_kostnad": 0.0
        }
