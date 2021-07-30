def custom_printer(string, log_to="log.log"):
    print(str(string))
    f = open(log_to, "a")
    f.write(str(string) + "\n")
    f.close()
