# masters-thesis-NER
Contains the codes developed for my master's thesis - Improving Named Entity Recognition

# Usage
## 1. Getting Started with Docker

This section will guide you through setting up and installing `Docker` on your machine. Docker allows you to package and run applications in isolated containers, ensuring consistency across different development and deployment environments.

### Installation :

### Windows
* **Download the <a href="https://docs.docker.com/desktop/install/windows-install/">Installer</a>:**
    - Follow the on-screen instructions to complete the installation. After installation, restart your computer.

### MacOS
* **Download the <a href="https://docs.docker.com/desktop/install/mac-install/">Installer</a> :**
  -  Follow the link and downloaded the installer. 
  - Open the downloaded .dmg file and drag the Docker icon to your Applications folder.
  - Open Docker from your Applications to complete the installation.


_Verify Installation:_ Open a terminal and type `docker --version` to see the Docker version.

Explore the Docker CLI with commands like docker info and docker help.
For detailed documentation and tutorials, visit the <a href="https://docs.docker.com">Docker Documentation</a> page.

## 2. Working with Docker and Docker Compose

This section provides basic commands to build your Docker container and manage multi-container applications using Docker Compose. For a more seamless development experience, it is recommended to use the Docker extension for Visual Studio Code, which allows you to manage Docker containers directly from the IDE.

### Building Docker Images
To build a Docker image from a Dockerfile, use the following command:
```rb
docker build -f Dockerfile .
```
This command builds a Docker image based on the instructions in your Dockerfile. The -f flag specifies the Dockerfile to use, and the . indicates that the build context is the current directory.

### Running Containers with Docker Compose
To start all services defined in your docker-compose.yml file, use:
```rb
docker-compose up
```
Add the -d flag if you prefer to run the containers in detached mode:
```rb
docker-compose up -d
```

This runs the containers in the background, allowing you to continue using the terminal while your containers are running. This mode is often used in production deployments or when you do not need to view the output directly in the terminal.

## 3. Setting Up Visual Studio Code
### Install Docker Extension:
Search for <a href="https://marketplace.visualstudio.com/items?itemName=ms-azuretools.vscode-docker">`Docker`</a> and install the Docker extension provided by Microsoft. This extension simplifies managing Docker images and containers from within the IDE.

### Attach to Container:
Once your container is up and running, open the Docker view in Visual Studio Code. In the Containers section, you will see your running containers. Right-click on the container you want to work with and select "Attach Visual Studio Code". This will open a new VS Code window connected to the container.

You can now run, edit, and debug your code directly inside the container, ensuring that your development environment matches your deployment environment closely.

By following these steps, you set up an efficient workflow for developing and testing Dockerized applications directly within Visual Studio Code.
