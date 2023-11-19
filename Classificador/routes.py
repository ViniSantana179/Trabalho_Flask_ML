from Classificador import app  
from flask import Flask, render_template,request,  url_for

@app.route('/', methods=['POST', 'GET'])
def home():
    if request.method == 'POST':
        _classificador =  request.form['classifier']
        print(_classificador)
        if _classificador == '1':
            _classificador = "DecisionTreeClassifier"
        elif _classificador == '2':
            _classificador = "RandomForestClassifier"
        elif _classificador == '3':
            _classificador = "MLPClassifier"
        elif _classificador == '4':
            _classificador = "KNeighborsClassifier"
        else:
            return render_template('index.html')
        return  render_template('index.html', classificador=_classificador)
    else:
        return render_template('index.html')
    

@app.route('/treinar/<int:classficador>', methods=['POST', 'GET'])
def treinar(classficador):
    from matplotlib import pyplot as plt
    from sklearn import tree
    from sklearn.metrics import accuracy_score
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    from sklearn.neural_network import MLPClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, f1_score, classification_report
    from random import randint
    import os

    if request.method == 'POST':

        att1 = int(request.form['att1']) 
        att2 = int(request.form['att2']) 
        att3 = int(request.form['att3']) 


        # Coleta e preparação
        iris = load_iris()
        X = iris.data
        y = iris.target

        # Divisão Treinamento/Classe
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state =0)

        # Escolher o Algoritmo

        pasta = 'E:/Topicos de Softwware/Flask_MachineLearning/Classificador/static/graficos'


        lista_arquivos = os.listdir(pasta)

        # Iterar sobre a lista e deletar cada arquivo
        for arquivo in lista_arquivos:
            caminho_arquivo = os.path.join(pasta, arquivo)
            try:
                if os.path.isfile(caminho_arquivo):
                    os.remove(caminho_arquivo)
                    print(f'Arquivo {arquivo} deletado com sucesso.')
            except Exception as e:
                print(f"Erro ao deletar {arquivo}: {str(e)}")


        if classficador == 1:
            clf_name = 'DecisionTreeClassifier'
            clf_dt = DecisionTreeClassifier(max_depth=att1, random_state=att2, max_leaf_nodes=att3)
            clf_dt.fit(X_train, y_train)
            acc = clf_dt.score(X_test, y_test)
            y_pred = clf_dt.predict(X_test)
        elif classficador == 2:
            clf_name = 'RandomForestClassifier'
            clf_rf = RandomForestClassifier( n_estimators=att1, max_depth=att2, max_leaf_nodes=att3)
            clf_rf.fit(X_train, y_train)
            acc = clf_rf.score(X_test, y_test)
            y_pred = clf_rf.predict(X_test)
        elif classficador == 3:
            clf_name = 'MLPClassifier'
            clf_mlp = MLPClassifier( hidden_layer_sizes=att1, random_state=att2, max_iter=att3)
            clf_mlp.fit(X_train, y_train)
            acc = clf_mlp.score(X_test, y_test)
            y_pred = clf_mlp.predict(X_test)
        elif classficador == 4:
            clf_name = 'KNeighborsClassifier'
            clf_knn = KNeighborsClassifier( n_neighbors=att1, leaf_size=att2, p=att3)
            clf_knn.fit(X_train, y_train)
            acc = clf_knn.score(X_test, y_test)
            y_pred = clf_knn.predict(X_test)
        else:
            return render_template('index.html')

        f1_macro = f1_score(y_test, y_pred, average='macro')
        cm = confusion_matrix(y_test, y_pred)
        classes = iris.target_names.tolist()
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
        disp.plot()
        id_img = randint(1, 1000000)
        arquivo = f'meu_grafico_{id_img}'
        plt.savefig(f'{pasta}/{arquivo}.png')
        nome_arquivo = f'{arquivo}.png'
        return render_template('results.html', accuracy=acc, f1_score= f1_macro, url_img=nome_arquivo, clf_name=clf_name)
        
    else:
        return render_template('index.html')