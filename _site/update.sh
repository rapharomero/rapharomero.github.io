git checkout source -- _site
rsync -a --delete _site/* .
git add *
git commit -m "update site"
git push
