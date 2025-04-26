cd python
rm -rf build
python setup.py bdist_wheel
pip uninstall glass -y
cd dist
ls | xargs pip install
