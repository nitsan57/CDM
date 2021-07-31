import os


def custom_printer(string):
    if os.environ['redirect']:
        log_to = os.environ['redirect']
    else:
        log_to = 'log.log'
    print(str(string))
    f = open(log_to, "a")
    f.write(str(string) + "\n")
    f.close()
