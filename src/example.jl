using Flux
using .Kolmogorov
using MLDatasets
using Statistics: mean
using Random

train_x, train_y = MLDatasets.MNIST.traindata(Float32)
test_x, test_y = MLDatasets.MNIST.testdata(Float32)

train_x ./= 255.0
test_x ./= 255.0

train_x = reshape(train_x, :, size(train_x, 3))
test_x = reshape(test_x, :, size(test_x, 3))

train_y = Flux.onehotbatch(train_y .+ 1, 1:10)
test_y = Flux.onehotbatch(test_y .+ 1, 1:10)

kan_model = Kolmogorov.KAN([784, 128, 64, 10]; grid_size=5, spline_order=3, scale_base=1.0, base_activation=sigmoid)

function loss(x::AbstractArray{Float32, 2}, y::AbstractArray)
    ŷ = kan_model(x; update_grid=false)  
    return Flux.crossentropy(ŷ, y)
end

optimizer = Flux.Adam()
epochs = 10
batch_size = 128

for epoch in 1:epochs
    Random.seed!(epoch)
    shuffled_indices = randperm(size(train_x, 2))
    shuffled_train_x = train_x[:, shuffled_indices]
    shuffled_train_y = train_y[:, shuffled_indices]
    
    for i in 1:batch_size:size(train_x, 2)
        batch_indices = i:min(i + batch_size - 1, size(train_x, 2))
        batch_x = shuffled_train_x[:, batch_indices]
        batch_y = shuffled_train_y[:, batch_indices]
        
        Flux.train!(loss, Flux.params(kan_model), [(batch_x, batch_y)], optimizer)
    end

    train_loss = loss(train_x, train_y)
    train_accuracy = mean(Flux.onecold(kan_model(train_x), 1:10) .== Flux.onecold(train_y, 1:10))
    println("Epoch $epoch: Training loss = $train_loss, Accuracy = $train_accuracy")
end

test_predictions = Flux.onecold(kan_model(test_x), 1:10)
test_accuracy = mean(test_predictions .== Flux.onehotargmax(test_y, 1:10))
println("Test Accuracy: $test_accuracy")


test_predictions = Flux.onecold(kan_model(test_x), 1:10)
test_accuracy = mean(test_predictions .== Flux.onecold(test_y, 1:10))
println("Test Accuracy: $test_accuracy")
