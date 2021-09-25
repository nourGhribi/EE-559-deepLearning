# Deep Learning Projects
## Project 1 – Classification, weight sharing, auxiliary losses
This  section  details  our  approach  to  compare  twodigits  of  a  pair  of  two-channel  images  from  the  MNIST  data.The  goal  of  this  project  is  to  compare  the  accuracy  of  differentarchitectures and assess the performance improvement that canbe achieved through models with shared weights and the use of anauxiliary loss. The performance of each of these different modelswill  be  estimated  on  a  test  data  through  10  rounds  where  bothdata  and  weight  initialization  are  randomized  at  each  trainin
  * `models.py` contains the class definition of the different models we implemented.
  * `helpers.py` contains a set of helper methods shared by the different models.
  * `test.ipynb` is a Jupyter notebook, containing our code for performing hyperparameter optimization via grid search.
  * `test.py` is a scrit which can be used to generate a random train and test dataset, and compare the performance of the three models, with different auxiliary loss weights. The models are pre-trained and are loaded from the `pickles` folder.

## Project 2 – Mini deep-learning framework
The objective of this project is to design a mini “deep learning framework” using only pytorch’s
tensor operations and the standard math library, hence in particular without using autograd or the
neural-network modules.
