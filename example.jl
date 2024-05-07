using .Kolmogorov
using Flux
using Flux: onehotbatch, onecold, crossentropy, DataLoader
using MLDatasets: MNIST
using Random
using Flux: params 

Random.seed!(1234)

train_x, train_y = MNIST.traindata()
test_x, test_y = MNIST.testdata()

function preprocess(img)
    img = Float32.(img) ./ 255.0  
    return reshape(img, 28 * 28)  
end

train_x = [preprocess(train_x[:, :, i]) for i in 1:size(train_x, 3)]
test_x = [preprocess(test_x[:, :, i]) for i in 1:size(test_x, 3)]

train_y = onehotbatch(train_y .+ 1, 1:10)
test_y = onehotbatch(test_y .+ 1, 1:10)

train_data = DataLoader((train_x, train_y), batchsize=64, shuffle=true)

model = KAN([28 * 28, 128, 64, 10]; grid_size=5, spline_order=3)

loss_fn(x, y) = crossentropy(model(x), y)
optimizer = Flux.ADAM()

num_epochs = 10
train_data = DataLoader((train_x, train_y), batchsize=64, shuffle=true)

for batch in train_data
    println(typeof(batch))  
    x, y = batch
    println(size(x), size(y)) 
end

# this doesnt work yet
for epoch in 1:num_epochs
    total_loss = 0.0
    num_batches = 0

    for (x, y) in train_data  
        loss_val = Flux.Optimise.update!(optimizer, params(model)) do
            loss_fn(x, y)
        end
        total_loss += loss_val
        num_batches += 1
    end

    train_loss = total_loss / num_batches
    println("Epoch $epoch: Training loss = $train_loss")
end


function accuracy(model, data)
    correct = 0
    total = 0
    for (x, y) in data
        total += size(x, 2)
        correct += sum(onecold(model(x)) .== onecold(y))
    end
    return correct / total
end

test_data = DataLoader((test_x, test_y), batchsize=64)
println("Test accuracy: ", accuracy(model, test_data))
