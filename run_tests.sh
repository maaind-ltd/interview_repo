cd DjangoProject
export TESTING=true
./manage.py test tests/
rm -rf efs
unset TESTING