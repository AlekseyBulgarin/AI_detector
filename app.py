from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def index():
    # При первом открытии probability нет, передаём None
    return render_template('index.html', user_text=None, probability=None, error=None)

@app.route('/check', methods=['POST'])
def check():
    text = request.form.get('user_text', '')
    
    if len(text.strip()) < 20:
        return render_template('index.html', 
                              user_text=text, 
                              probability=None, 
                              error="Слишком короткий текст (минимум 20 символов)")
    
    # TODO: здесь потом добавишь вызов детектора
    # пока заглушка
    probability = 50  # временно, потом уберёшь
    
    return render_template('index.html', 
                          user_text=text, 
                          probability=probability, 
                          error=None)

if __name__ == '__main__':
    app.run(debug=True)