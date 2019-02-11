
#!/bin/bash
# To save typing
git checkout mac
git pull
git checkout dev
git merge mac
git push

git checkout server
git merge mac
git push
