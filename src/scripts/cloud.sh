# This needs to be copy pasted in the CMD override before creating the pod
# Copy the ssh key needed and set the correct permission
cp /workspace/ssh_keys/id_ed25519 ~/.ssh/
chmod 0600 ~/.ssh/id_ed25519

# Add the host
ssh-keyscan github.com >> ~/.ssh/known_hosts

# Prompt the user for the Git repo URL
read -p "Enter the SSH Git repository URL (e.g., git@github.com:user/repo.git): " REPO_URL

# Extract the repo name (strip .git at the end if present)
REPO_NAME=$(basename -s .git "$REPO_URL")

# Clone the repo
mkdir -p ~/Developer/Projects/
cd ~/Developer/Projects/
git clone "$REPO_URL"

git config --global pull.rebase true
git config --global user.name "Sri Hari Karthick"
git config --global user.email "sriharikarthik2641999@gmail.com"

# Cd and run the setup for installing dependencies
cd "$REPO_NAME"
# Update pip
python -m pip install --upgrade pip
# Remove old version of pytorch
pip uninstall torch torchvision torchaudio torchtext -y
# Install all the needed depdendencies
pip install -r requirements.txt