FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-devel

# copy files
ADD scripts /workspace/
RUN chmod +x /workspace/*.sh
RUN mkdir /mnt/data
RUN mkdir /mnt/pred
RUN pip install nibabel
