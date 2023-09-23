import socket
import sys

server_address = ('127.0.0.1', 8081)
node_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    node_socket.connect(server_address)
except (ConnectionRefusedError, ConnectionResetError, ConnectionError, ConnectionAbortedError, Exception):
    print("Connection refused by the server. Exiting...")
    sys.exit(1)

while True:
    try:
        message = node_socket.recv(2**15).decode()
        if not message:
            print("Connection closed by the server.")
            break
        print(f"Received broadcast message: {message}")
        if "Enter 1 to accept or 0 to reject the block:" in message:
            response = input()
            node_socket.send(response.encode())
    except Exception as e:
        print(f"Error: {str(e)}")
        break
