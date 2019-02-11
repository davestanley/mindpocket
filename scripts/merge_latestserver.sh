
#!/bin/bash
# To save typing

# Pull everything
# git co dev
# git pull
# git co mac
# git pull
# git co server
# git pull

# Merge dev branch everywhere
git checkout server
git pull

git checkout mac
git merge server
git push

git checkout dev
git merge server
git push

# git checkout server
