pydoctor \
    --project-name=iislib \
    --project-url=https://github.com/franchesoni/iis_framework/ \
    --make-html \
    --html-output=docs/api \
    --project-base-dir="." \
    --docformat=numpy \
    --theme=base \
    --intersphinx=https://docs.python.org/3/objects.inv \
    ./iislib
