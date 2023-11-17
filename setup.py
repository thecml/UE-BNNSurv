from setuptools import setup, find_packages

setup(
    name = 'bnnsurv', 
    version = '0.1.3',
    description = 'TensorFlow 2.x Bayesian Neural Network for Survival Analysis',
    packages = find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    author = 'Christian Marius Lillelund',
    author_email = 'chr1000@gmail.com',
    
    long_description = open('README.md').read(),
    long_description_content_type = "text/markdown",

    url='https://github.com/thecml/UE-BNNSurv',

    classifiers  = [
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        "License :: OSI Approved :: MIT License",
        'Intended Audience :: Developers',
        'Intended Audience :: Other Audience',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Operating System :: OS Independent',
    ],
    
    install_requires = [
        'numpy ~= 1.22',
        'scikit-survival ~= 0.22',
        'tensorflow >= 2.11',
        'tensorflow-probability >= 0.19'
    ],
    
    keywords = ['Deep Learning', 'Neural Network', 'Bayesian Learning', 'Survival Analysis'],
    
)
