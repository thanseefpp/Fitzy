from views import server

################################### EXECUTE APPLICATION #################################

if __name__ == "__main__":
    server.run(host='0.0.0.0', port=8000,debug=True)