# if you wanna run the ai run "pip install -r requirements.txt" and run "py JokeAi.py" and you should see a joke
# you can change the title in in the bottom in "JokeAi.py" and the trained model  at line 61, but if you change the model you also so need to change the title_filt and body_filt at line 32 and line 38 following to the respective model, see below
 "title_generator_test.h5" is a low trained ai on a large data set. If you want to use this change title and body filt to 30000
"title_generator_test_overFit.h5" is a medium trained ai on a samller dataset. If you want to use this change title and body fillt to 3000
 "title_generator_test_extra_over_fit.h5" is a overtrained ai on a small dataset, theres many word this ai havent seen so its pretty bad, but if you wanna test it change title and body filt to 300.

# If you train your own ai run tesat.py, and you can change bvat size, epoch and training data in the tesat.py file
