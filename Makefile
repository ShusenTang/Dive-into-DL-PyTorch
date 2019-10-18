wwwdocs:
	bash script/prepare_wwwdocs.sh

clean:
	find . -name '*.pyc' -type f | xargs rm -rf
	rm -rf .wwwdocs
