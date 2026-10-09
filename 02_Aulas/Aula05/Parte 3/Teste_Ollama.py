# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 04:02:59 2026

@author: rodri
"""

import requests

resposta = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "qwen2.5:7b-instruct-q4_K_M",
        "prompt": "O que é um alagamento urbano?",
        "stream": False
    },
    timeout=180
)

resposta.raise_for_status()
print(resposta.json()["response"])