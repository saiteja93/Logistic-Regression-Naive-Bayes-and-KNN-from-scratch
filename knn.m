
train = loadMNISTImages("train-images.idx3-ubyte"); 
%For the sake of better understanding I transposed the matr
train_transposed = transpose(train);
train_labels = loadMNISTLabels("train-labels.idx1-ubyte");

test_images = loadMNISTImages("t10k-images.idx3-ubyte");
test_images_transposed = transpose(test_images);
test_labels = loadMNISTLabels("t10k-labels.idx1-ubyte");

neighbors = [100];

train_data_length = length(train);
test_data_length = length(test_images);


%nearest_neighbors = zeros(neighbors(1),2)
nearest_neighbors = zeros(train_data_length,2);
max_nearest_neighbors = zeros(train_data_length,2);
predicted_labels = zeros(test_data_length,2);

num_neighbor_list = [1 3 5 10 30 50 70 80 90 100];

%First we find the distance matrix of each point from all the other points
all_nearest = zeros(test_data_length, 100);
ex_list = [100];
%pl = zeros(10,1)
for i = 1:test_data_length
    for j = 1:60000
        eu_dist = 0;
        for k=1:784
            eu_dist = eu_dist + (test_images(k,i)-train(k,j))^2;
        end

        eu_dist = sqrt(eu_dist);
        
        nearest_neighbors(j,1) = train_labels(j,:);
        nearest_neighbors(j,2) = eu_dist;

    end
    % we sort the distances and pick the smallest 100 classlabels for each
    % training row.
    max_nearest_neighbors = sortrows(nearest_neighbors,2);
    all_nearest(i,:) = max_nearest_neighbors(1:100,1);
    %pl(i,1) =  mode(nearest_neighbors(:,1)) 
end




p_l = zeros(test_data_length, 10);
for i=1:test_data_length
    for k =1:length(num_neighbor_list)
        %We use the mode functionality of Matlab to calculate the majority
        %class label in the given k neighbors
        p_l(i,k) = mode(all_nearest(i,1:num_neighbor_list(k)));
    end

end

accuracy = zeros(length(num_neighbor_list),2);

for k =1:length(num_neighbor_list)
    count = 0;
    for i=1:test_data_length
        if p_l(i,k) == test_labels(i)
            count = count+1;
        end
    end
    %Finally populating the accuracy matrix with k values and the
    %corresponding accuracy values in each row.
    accuracy(k,2) = count/test_data_length;
    accuracy(k,1) = num_neighbor_list(k);
end
plot(accuracy(:,1),accuracy(:,2))

%eu_dist = euclideanDistance(test, train)