# Luuma API Server

The Luuma API server is a rust application that provides a RESTful API for the Luuma Chat Client. The server is built using the Rocket web framework and is designed to be run on a Linux server. The server is responsible for relaying messages between clients and the selected LLM provider. It is also possible to direct to a self-hosted LLM server.

## Installation

Create new user for server
sudo adduser --system --group --disabled-login --no-create-home --shell /bin/false luuma_api_user

sudo chown -R luuma_api_user:luuma_api_user /www/luuma-api/
sudo chmod -R 750 /www/luuma-api/
sudo chmod -R 770 /www/luuma-api/logs/


sudo nano /etc/systemd/system/luuma_api.service

sudo cp luuma_api.service /etc/systemd/system/luuma_api.service

sudo systemctl daemon-reload
sudo systemctl enable luuma_api.service
sudo systemctl start luuma_api.service
