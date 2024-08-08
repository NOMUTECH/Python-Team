from models import get_openai_response, get_google_response
from flask import Flask, request, jsonify, redirect
from flask_swagger_ui import get_swaggerui_blueprint


app = Flask(__name__)
default_prompt = """You are an AI working within the FIST application, a powerful Super App designed specifically for medium to large-scale manufacturing businesses. FIST serves as the smartest tracking assistant tailored to each company. It helps businesses take control in their own production areas and within their industry, providing flexible two-phase functionality. Your goal is to assist these businesses in reaching desired manufacturers or vendors more efficiently, ultimately empowering them in the marketplace."""
def send_to_all_models(
        prompt,
        system_prompt
        ):
    responses = {}
    
    # OpenAI model
    try:
        responses['OpenAI'] = get_openai_response(prompt, system_prompt)
    except Exception as e:
        responses['OpenAI'] = f"Error: {str(e)}"
    
    # Google model
    try:
        responses['Google'] = get_google_response(prompt, system_prompt)
    except Exception as e:
        responses['Google'] = f"Error: {str(e)}"
    
    return responses

@app.route('/api/get_responses', methods=['POST'])
def get_responses():
    global default_prompt
    data = request.json
    prompt = data.get('prompt')

    if data.get("system_prompt"):
        system_prompt = data.get('system_prompt')
    else:
        system_prompt = default_prompt
    results = send_to_all_models(prompt, system_prompt)
    return jsonify(results)




@app.route('/')
def redirect_to_docs():
    return redirect('/docs')




SWAGGER_UI_SCRIPT = "https://cdn.jsdelivr.net/npm/swagger-ui-dist@latest/swagger-ui-bundle.js"
SWAGGER_UI_STYLE = "https://cdn.jsdelivr.net/npm/swagger-ui-dist@latest/swagger-ui.css"

swaggerui_blueprint = get_swaggerui_blueprint(
    '/docs',
    '/static/swagger.json',
    config={
        'app_name': "My Application"
    }
)

app.register_blueprint(swaggerui_blueprint)

if __name__ == "__main__":
    app.run(debug=True)
