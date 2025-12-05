.EXPORT_ALL_VARIABLES:
.ONESHELL: # Applies to every targets in the file!


install_openspec:
	# Download and install nvm:
	curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash

	# in lieu of restarting the shell
	. "$$HOME/.nvm/nvm.sh"

	# Download and install Node.js:
	nvm install 24

	# Verify the Node.js version:
	node -v # Should print "v24.11.1".

	# Verify npm version:
	npm -v # Should print "11.6.2".

	npm install -g @fission-ai/openspec@latest
