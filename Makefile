readme-update:
	curl https://raw.githubusercontent.com/WangRongsheng/awesome-LLM-resourses/main/README.md | awk -f riss.awk >content/readme-in-static-site.md
