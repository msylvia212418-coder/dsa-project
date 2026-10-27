
import socket

HOST = "0.0.0.0"
PORT = 5000


def start_receiver():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((HOST, PORT))
        server.listen(1)

        print("=" * 40)
        print(" NETWORK ROUTING - RECEIVER")
        print("=" * 40)
        print(f"Listening on port {PORT}...")
        print("Waiting for sender to connect...\n")

        conn, address = server.accept()

        with conn:
            print(f"Connected to {address[0]}:{address[1]}")
            print("Waiting for messages. Ctrl+C to stop.\n")

            with conn.makefile("r", encoding="utf-8") as stream:
                for line in stream:
                    message = line.rstrip("\r\n")
                    print(f"Received message: {message}")

        print("\nSender disconnected.")


if __name__ == "__main__":
    try:
        start_receiver()
    except OSError as error:
        print(f"Receiver error: {error}")
    except KeyboardInterrupt:
        print("\nReceiver stopped.")
