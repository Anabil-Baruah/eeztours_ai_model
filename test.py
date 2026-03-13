import google.generativeai as genai

genai.configure(api_key="AIzaSyASZ-m02Q77LO51oKJg81xUlmp73ABvVTc")

for m in genai.list_models():
    if "generateContent" in m.supported_generation_methods:
        print(m.name)