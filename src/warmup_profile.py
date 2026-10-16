"""Synthetic, fixed support-chat workload used only by X-CR-Warmup requests.

The payload exercises prefill at approximately the real CreaRápido context size
and at least 16 decode steps. It contains NO real chats, customer data, or keys.
The exact prompt-token count depends on the baked Qwen tokenizer; the worker
records actual prompt_tokens from vLLM usage to verify the approximation.
"""
from __future__ import annotations

import os

WARMUP_PROFILE = "support-chat-v2"

_SUPPORT_GUIDE = (
    "DOCUMENTACIÓN SINTÉTICA DE SOPORTE CREARÁPIDO (SOLO PRUEBA, NO DATOS REALES). "
    "La plataforma permite crear una página web describiendo un negocio y editando el resultado. "
    "El usuario puede cambiar textos, imágenes, botones, secciones y páginas en un editor visual. "
    "Para publicar, revisa el contenido y comprueba los ajustes disponibles en su panel. "
    "Si pregunta por un precio, límite o función concreta, consulta primero la documentación oficial. "
    "No prometas que una función existe ni inventes un procedimiento si no está confirmada. "
    "Para un catálogo, organiza productos, títulos, descripciones, fotos y categorías pertinentes. "
    "Para reservas, explica cómo localizar las opciones documentadas y validar los datos del negocio. "
    "Para SEO, sugiere títulos claros, descripciones útiles y estructura semántica de encabezados. "
    "Para una página de contacto, indica cómo revisar el formulario, correo y llamadas a la acción. "
    "Si alguien necesita asistencia humana, explica los canales oficiales sin inventar disponibilidad. "
    "En un error del editor, pregunta qué intentó hacer, dónde ocurrió y qué mensaje recibió. "
    "Para sitios bilingües, recomienda revisar todos los textos y enlaces en cada idioma. "
    "Mantén las respuestas breves, amables, en el idioma del último mensaje y con pasos concretos. "
)

# Approximately 1,900 Qwen prompt tokens; verify using usage.prompt_tokens
# because text character count is not the same as model token count.
_SYNTHETIC_KNOWLEDGE = (_SUPPORT_GUIDE * 6)[:8000]

_HISTORY = [
    {"role": "user", "content": "Hola, quiero crear una página para mi negocio."},
    {"role": "assistant", "content": "Con gusto, dime el nombre y una breve descripción de tu negocio."},
    {"role": "user", "content": "Quiero mostrar mis servicios y un catálogo sencillo."},
    {"role": "assistant", "content": "Puedes empezar con secciones de servicios, catálogo y contacto."},
    {"role": "user", "content": "¿Y puedo editar textos e imágenes después?"},
    {"role": "assistant", "content": "Sí, puedes revisar esos elementos en el editor visual."},
    {"role": "user", "content": "¿Cómo organizo las secciones?"},
    {"role": "assistant", "content": "Da prioridad a una portada clara y después servicios y contacto."},
    {"role": "user", "content": "¿Cómo puedo mejorar los títulos?"},
    {"role": "assistant", "content": "Escribe títulos concretos que expliquen el valor de cada sección."},
]


def warmup_output_tokens() -> int:
    value = int(os.getenv("WARMUP_OUTPUT_TOKENS", "24"))
    if not 16 <= value <= 32:
        raise ValueError("WARMUP_OUTPUT_TOKENS must be between 16 and 32")
    return value


def synthetic_support_warmup(model_id: str) -> dict:
    max_tokens = warmup_output_tokens()
    return {
        "model": model_id,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Eres un asistente de soporte técnico para un creador de páginas web. "
                    "Este es un contexto completamente ficticio usado para precalentar GPU; "
                    "no es una conversación real. Responde en español.\n\n"
                    + _SYNTHETIC_KNOWLEDGE
                ),
            },
            *_HISTORY,
            {
                "role": "user",
                "content": (
                    "Dame una explicación completa, enumerada y práctica para organizar, "
                    "personalizar y publicar un sitio web. Incluye consejos para el catálogo "
                    "y para mejorar la legibilidad de la página."
                ),
            },
        ],
        "max_tokens": max_tokens,
        "min_tokens": 16,
        "ignore_eos": True,
        "temperature": 0,
        "stream": False,
    }


def warmup_context_chars() -> int:
    return len(_SYNTHETIC_KNOWLEDGE)
