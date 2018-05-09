# Vectors
#--------------------------------------------------
# creating the vector using the c() function
# the c() is concatinate function
#--------------------------------------------------
vect <- c(1, 3, 5, 5, 7)

# Rrepresent the type and class of the object
class(vect)
typeof(vect)

vect_car <- c("a", "b", "c")

# get the lenght of the vector
length(vect)
#--------------------------------------------------
# vector using the colon oprator
#--------------------------------------------------
x <- 1:5

w <- c(1:9, 70)
#--------------------------------------------------
# Creating the vector using the rep() function
#--------------------------------------------------
rep_x <- rep(4, 6)
rep_x

#--------------------------------------------------
# Creating the vector using the seq() function
#--------------------------------------------------
x <- seq(from = 1, to=6 , by = 0.5)
x
# you can easily do this without specifing the from and to
y <- seq(0, 20, 2)
y
# In reverse direction
z <-  seq(10, 1, -2)
z
# creating the sequence of vector using length function
k <- seq(1, 5, length = 10)
k
#--------------------------------------------------
# Creating the vector of random numbers
#--------------------------------------------------
# sample() will create the vector of random numbers
vect1 <- sample(9:100, 9)
vect1
# create the vector with replacement( Duplication of the records ) 
vect2 <- sample(30, 20, replace = TRUE)
vect2
#--------------------------------------------------
# Accessing the component using the indices
#--------------------------------------------------
vect2[2] # simple indices
vect2[c(2,4,6)] # with specific indices
vect2[2:8] # Accessing in order ascending
vect2[8:2] # Accessing in reverse order
# Accessing the component without some indices
vect2[-2]
vect2[-2:-10]

#--------------------------------------------------
# Editing and adding the components
#--------------------------------------------------
vect <- seq(10, 100, length = 10)
vect
# Adding an component at the end
vect[11] <- 110
vect
# Editing the component 
vect[3] <- 33
vect

