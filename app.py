import math
from flask import Flask, render_template, request, jsonify
from sympy import Symbol, diff, sympify, lambdify
import pandas as pd

# Inicializa o app Flask
app = Flask(__name__)

# --- Lógica do Método de Newton-Raphson ---
# (Baseado no seu script newton_raphson.py)
def calcular_newton(f_str, x0_val, erro_val, k_val):
    resultados = {
        "i": [], "Xi": [], "Xi+1": [],
        "[(X(i+1)-Xi)/X(i+1)] x 100%": []
    }
    
    x = Symbol('x')
    
    try:
        # 'sympify' transforma a string (ex: "x**2 - 2") em uma expressão SymPy
        f = sympify(f_str)
        f_linha = diff(f, x) # Calcula a derivada
    except Exception as e:
        return f"Erro ao processar a função: {e}"

    xi_anterior = x0_val
    for i in range(k_val):
        resultados["i"].append(i)
        resultados["Xi"].append(float(xi_anterior))
        
        # Avalia f(x) e f'(x) no ponto xi_anterior
        f_val = f.subs(x, xi_anterior)
        f_linha_val = f_linha.subs(x, xi_anterior)
        
        if abs(f_linha_val) < 1e-10:
            return "Derivada zero ou muito próxima de zero. O método falhou."
            
        # Cálculo do Newton-Raphson
        xi_futuro = xi_anterior - (f_val / f_linha_val)
        resultados["Xi+1"].append(float(xi_futuro))
        
        # Cálculo do erro
        if i == 0:
            erro = 100.0  # Primeira iteração sempre tem erro alto
        else:
            if abs(xi_futuro) < 1e-10:
                erro = 100.0
            else:
                erro = float(abs(xi_futuro - xi_anterior) / abs(xi_futuro) * 100)
        
        resultados["[(X(i+1)-Xi)/X(i+1)] x 100%"].append(erro)
        
        # Verifica convergência (não na primeira iteração)
        if i > 0 and erro < erro_val:
            break
        
        xi_anterior = xi_futuro
        
    df = pd.DataFrame(resultados).set_index('i')
    # Formata números com 6 casas decimais
    df = df.round(6)
    # Converte o DataFrame para HTML para ser exibido na página
    return df.to_html(classes="table table-striped", border=0)


# --- Lógica do Método da Bisseção ---
# (Baseado no seu script bissecao.py)
def calcular_bissecao(f_str, a_val, b_val, erro_val, k_val):
    resultados = {
        "i": [], "a": [], "b": [], "Xi": [],
        "f(a)": [], "f(b)": [], "f(Xi)": [],
        "|X(i+1) - Xi|/X(i+1)": []
    }
    
    x = Symbol('x')
    try:
        f = sympify(f_str)
        # 'lambdify' transforma a expressão SymPy em uma função Python rápida
        func = lambdify(x, f, 'math')
    except Exception as e:
        return f"Erro ao processar a função: {e}"
    
    # Validação: verifica se f(a) e f(b) têm sinais opostos
    fa_inicial = func(a_val)
    fb_inicial = func(b_val)
    
    if fa_inicial * fb_inicial > 0:
        return "Erro: f(a) e f(b) devem ter sinais opostos para o método da bisseção funcionar!"

    xi_anterior = None
    for i in range(k_val):
        xi = (a_val + b_val) / 2
        fa = func(a_val)
        fb = func(b_val)
        fxi = func(xi)
        
        resultados["i"].append(i + 1)
        resultados["a"].append(a_val)
        resultados["b"].append(b_val)
        resultados["Xi"].append(xi)
        resultados["f(a)"].append(fa)
        resultados["f(b)"].append(fb)
        resultados["f(Xi)"].append(fxi)
        
        # Cálculo do erro
        if xi_anterior is None:
            erro_calc = 100.0  # Primeira iteração
        else:
            if abs(xi) < 1e-10:
                erro_calc = 100.0
            else:
                erro_calc = float(abs(xi - xi_anterior) / abs(xi)) * 100
        
        resultados["|X(i+1) - Xi|/X(i+1)"].append(erro_calc)
        
        # Verifica convergência (não na primeira iteração)
        if xi_anterior is not None and erro_calc < erro_val:
            break
        
        # Atualiza intervalo
        if fxi * fa < 0:
            b_val = xi
        elif fxi * fa > 0:
            a_val = xi
        elif abs(fxi) < 1e-10:  # Encontrou a raiz exata
            break
            
        xi_anterior = xi

    df = pd.DataFrame(resultados).set_index('i')
    # Formata números com 6 casas decimais
    df = df.round(6)
    return df.to_html(classes="table table-striped", border=0)


# --- Definição das Rotas (Páginas) ---

# Rota para o Menu Principal
@app.route("/")
def index():
    return render_template("index.html")

# Rota para a página da Bisseção
@app.route("/bissecao")
def page_bissecao():
    return render_template("bissecao.html")

# Rota para a página do Newton-Raphson
@app.route("/newton")
def page_newton():
    return render_template("newton.html")

# --- Rotas de Cálculo (API) ---

@app.route("/calcular_bissecao", methods=["POST"])
def api_bissecao():
    data = request.json
    try:
        f_str = data["fx"]
        a = float(data["a"])
        b = float(data["b"])
        erro = float(data["erro"])
        iteracoes = int(data["iteracoes"])
        
        resultado_html = calcular_bissecao(f_str, a, b, erro, iteracoes)
        return jsonify({"tabela_html": resultado_html})
    except Exception as e:
        return jsonify({"erro": str(e)})

@app.route("/calcular_newton", methods=["POST"])
def api_newton():
    data = request.json
    try:
        f_str = data["fx"]
        x0 = float(data["x0"])
        erro = float(data["erro"])
        iteracoes = int(data["iteracoes"])
        
        resultado_html = calcular_newton(f_str, x0, erro, iteracoes)
        return jsonify({"tabela_html": resultado_html})
    except Exception as e:
        return jsonify({"erro": str(e)})

# Roda o aplicativo
if __name__ == "__main__":
    app.run(debug=True)