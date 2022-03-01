rm -rf *
git checkout main -- _site
cp -r ./_site/* .
rm -r _ste
git add *
git commit -m "update site"
git push
