Disclaimer: Most of the code and documentation, including this readme, has been written by my input, chatgpt's output, and so on.

PuttyGPT - The Conversational AI Platform
=========================================

# 🚧 Before you proceed 🚧
Please note that this is a very early version of the project, and we are still in the process of wrapping things up, wiring components together, and experimenting with new ideas. As such, the project may undergo significant changes and updates as we continue to evolve and refine our vision. Any ideas are welcome and I will get back to you as soon as I can.
[CONTRIBUTING.md](./CONTRIBUTING.md)

PuttyGPT is a conversational AI project powered by OpenAI's GPT-4, Weaviate for vector storage, and other state-of-the-art tools, providing a comprehensive and user-friendly interface for developers, AI enthusiasts, and business professionals. By utilizing the latest technologies and with the collaboration of our package contributors, we aim to create a solid foundation for diverse AI applications.

:sparkles: Features
-------------------

- Powered by OpenAI's GPT-4 for natural language understanding and generation
- Utilizes SearxNG for search capabilities and Weaviate for vector storage and search
- Supports a range of AI tools, including summarization, sentiment analysis, and OpenAPI integration
- Customizable prompt templates for diverse use cases
- Efficient concurrent task execution using asyncio and aiohttp
- Detailed tracing and callback mechanisms for monitoring and debugging
- Designed for extensibility and easy integration with other APIs and services
- Dockerized deployment for ease of installation and scalability



:rocket: Getting Started
------------------------

### Docker Installation

Clone the repository and navigate to the project directory:

```
git clone https://github.com/yourusername/puttygpt.git
cd puttygpt

```
(Put this in init.sh later or makefile)
Build and run the Docker containers using the provided docker-compose files:

```
# Prerequisite (I can probably get rid of this but meh not right now)
docker network create weaviate-network

# Without public grpc endpoint
docker-compose -f docker-compose.weaviate.yml -f docker-compose.yml up --build

# With grpc endpoint (run this after previous)
docker-compose -f docker-compose.weaviate.yml -f docker-compose.yml -f docker-compose.brazen.yml up brazen --build

# Copy env file
cp .env.example .env

# replace the string for searxng
sed -i "s|ReplaceWithARealKey\!|$(openssl rand -base64 33)|g" .env

```

Cleanup:
```
docker-compose -f docker-compose.weaviate.yml -f docker-compose.yml -f docker-compose.brazen.yml down --remove-orphans
```


### Usage

To interact with the application, monitor your docker logs for `eve`

:wrench: Customization
----------------------

PuttyGPT offers customization options for developers and businesses to tailor the AI experience to their specific needs. You can create your own agents and tools, modify the prompt templates, and even integrate with external APIs and services to unlock new possibilities.

:bulb: Possible Future Applications
-----------------------------------

By leveraging our technology stack, PuttyGPT has the potential to enable a variety of innovative applications in the future:

- Virtual assistants for personal or professional use
- Intelligent document automation and processing
- Real-time market analysis and data-driven decision making
- Rapid prototyping and idea generation
- Integration with various APIs and services for extended functionality

:balance\_scale: Legal and Ethical Considerations
-------------------------------------------------

We are committed to the responsible use of AI and encourage users to consider legal and ethical implications when using PuttyGPT. Please ensure that your use of PuttyGPT adheres to applicable laws and regulations and respects the rights of others, including privacy and intellectual property rights.

:handshake: Contributing
------------------------

We welcome contributions to PuttyGPT! Whether you're a developer, AI enthusiast, or a business professional, your ideas and expertise can help make this project even better. We also want to extend our gratitude to the package contributors for their incredible work. Check out our [CONTRIBUTING.md](./CONTRIBUTING.md) for more information on how to contribute.

:memo: License
--------------

This project is licensed under the [MIT License](./LICENSE).

