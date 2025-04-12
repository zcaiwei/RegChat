from setuptools import setup, find_packages

setup(
    name='regchat',
    version='v1.0.0',
    author='Caiwei Zhen',
    author_email='cwzhen@whu.edu.cn',
    description='RegChat: leveraging multi-omics data for modeling and inference of intercellular communications and signaling pathways',
    long_description='RegChat is an advanced Python package designed for CCC inference of single-cell multi-omics datasets. It employs graph-regularized neural networks and heterogeneous graph attention neural networks to dissect multi-layer signaling transduction from ligands to TGs, providing valuable insights into cellular behaviors in complex biological processes.',
    long_description_content_type='text/markdown',
    url='https://github.com/zcaiwei/RegChat',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3.10.9',
    ],
    python_requires='>=3.10',
    install_requires=[
        'anndata==0.10.6',
        'h5py==3.7.0',
        'torch==2.2.1',
        'jupyterlab==3.5.3',
        'matplotlib==3.7.0',
        'numpy==1.24.4',
        'pandas==1.5.3',
        'scikit-learn==1.2.1',
        'scipy==1.10.1',
        'scanpy==1.9.6',
        'scikit-misc==0.5.1'
    ],
)
