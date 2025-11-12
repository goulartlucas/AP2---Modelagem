import math
from flask import Flask, render_template, request, jsonify
from sympy import Symbol, diff, sympify, lambdify
import pandas as pd
import io

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
        
        if f_linha_val == 0:
            return "Derivada zero. O método falhou."
            
        # Cálculo do Newton-Raphson
        xi_futuro = xi_anterior - (f_val / f_linha_val)
        resultados["Xi+1"].append(float(xi_futuro))
        
        if i == 0:
            resultados["[(X(i+1)-Xi)/X(i+1)] x 100%"].append(100.0)
        else:
            erro = float(abs(xi_futuro - xi_anterior) / xi_futuro * 100)
            resultados["[(X(i+1)-Xi)/X(i+1)] x 100%"].append(erro)
            if erro < erro_val:
                break
        
        xi_anterior = xi_futuro
        
    df = pd.DataFrame(resultados).set_index('i')
    # Converte o DataFrame para HTML para ser exibido na página
    return df.to_html(classes="table table-striped", border=0)


# --- Lógica do Método da Bisseção ---
# (Baseado no seu script bissecao.py)
def calcular_bissecao(f_str, a_val, b_val, erro_val, k_val):
    resultados = {
        "i": [], "a": [], "b": [], "Xi": [],
        "f(a)": [], "f(b)": [], "f(Xi)": [],
        "|X(i+1) - Xi|/X(i+1)": [1]
    }
    
    x = Symbol('x')
    try:
        f = sympify(f_str)
        # 'lambdify' transforma a expressão SymPy em uma função Python rápida
        func = lambdify(x, f, 'math')
    except Exception as e:
        return f"Erro ao processar a função: {e}"

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
        
        erro_calc = 1.0 # Valor default
        
        if i > 0:
            xi_anterior = resultados["Xi"][-2] # Pega o Xi da iteração anterior
            erro_calc = float(abs(xi - xi_anterior) / xi) * 100
            resultados["|X(i+1) - Xi|/X(i+1)"].append(erro_calc)
        
        if erro_calc < erro_val and i > 0:
            break
            
        if fxi * fa < 0:
            b_val = xi
        elif fxi * fa > 0:
            a_val = xi
        elif fxi == 0:
            break
            
    # Remove o '1' inicial se mais de uma iteração ocorreu
    if len(resultados["i"]) > 0:
        resultados["|X(i+1) - Xi|/X(i+1)"][0] = resultados["|X(i+1) - Xi|/X(i+1)"][1] if len(resultados["i"]) > 1 else 0.0


    df = pd.DataFrame(resultados).set_index('i')
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