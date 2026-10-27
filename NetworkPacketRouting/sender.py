
import socket

HOST = "127.0.0.1"  # Local test: this same laptop
PORT = 5000


def start_sender():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
        client.connect((HOST, PORT))

        print("=" * 40)
        print(" NETWORK ROUTING - SENDER")
        print("=" * 40)
        print("Connected to receiver!")
        print("Type a message and press Enter.")
        print("Type /quit to disconnect.\n")

        with client.makefile("w", encoding="utf-8") as stream:
            while True:
                message = input("You: ")

                if message.lower() == "/quit":
                    break

                stream.write(message + "\n")
                stream.flush()

        print("Disconnected from receiver.")


if __name__ == "__main__":
    try:
        start_sender()
    except ConnectionRefusedError:
        print("Connection refused. Start receiver.py first.")
    except OSError as error:
        print(f"Sender error: {error}")
    except KeyboardInterrupt:
        print("\nSender stopped.")
