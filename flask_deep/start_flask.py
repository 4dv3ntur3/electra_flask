from electra_server import app


if __name__ == "__main__":

    # app.run(host='127.0.0.1', use_reloader=False) # service

    app.run(host='127.0.0.1') # develop 

    # 실행 시킬 때 flask run --no-reload 
