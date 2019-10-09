all: checkcode

tag:
	bash script/addtag.sh

urgency:
	export URGENCY_SUPPORT="urgency" && bash script/addtag.sh

check:
	bash script/check_code.sh

checkcode:
	bash script/check_code.sh 2>&1 | grep -v "line is too long" | grep -v "OK" | grep -v "warning" | grep "code/"

start:
	bash entry.sh start

reload:
	bash entry.sh reload

stop:
	bash entry.sh stop

clean:
	rm -rf logs/*.log*
	rm -rf *_temp
