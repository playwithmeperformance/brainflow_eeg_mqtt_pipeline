#!/usr/bin/python3
from pythonosc.udp_client import SimpleUDPClient

ip = "127.0.0.1"
port = 6010

client = SimpleUDPClient(ip, port)  # Create client

client.send_message("/ctrl", ['hello', 10.0])   # Send float message
# client.send_message("/some/address", [1, 2., "hello"])  # Send message with int, float and string
