import socket

def check_port(host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(3)
        result = s.connect_ex((host, port))
        return result == 0

print(check_port("192.168.200.130", 3306))