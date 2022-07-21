%other params bounds
global E1_low E1_high E2_low E2_high yield_low yield_high;
E1_low = 2.5;
E1_high = 7.5;
E2_low = 0.5;
E2_high = 6.5;
yield_low = 0.1;
yield_high = 1.3;

load('prior_samples.mat')
% assume a uniform prior on parameters, generate prior
%sample_size = 100000;
%randm = rand(sample_size,3);
%randm(:,1) = randm(:,1)*(yield_high-yield_low) + yield_low;
%randm(:,2) = randm(:,2)*(E1_high-E1_low) + E1_low;
%randm(:,3) = randm(:,3)*(E2_high-E2_low) + E2_low;
%theta_prior = [];
%for i = 1:sample_size
%    if randm(i,2) > randm(i,3)
%        theta_prior = [theta_prior;randm(i,:)];
%    end
%end
disp(length(theta_prior))

%%
modelfile = 'model1.json';
weights = 'model1.h5';
forward_NN = importKerasNetwork(modelfile,'WeightFile',weights,'OutputLayerType','regression');
actions_array = 1.6:0.02:2;

%% greedy algorithm
theta = [0.1, 7, 5];
expected_depths_by_actions = compute_expected_depths_by_actions(theta_prior, actions_array, forward_NN);
depths_by_actions = compute_depths_by_actions(theta, actions_array, forward_NN);

theta_prior1 = theta_prior;
entropy0 = entropy_calculation(theta_prior1)
for i=1:3
    entropy_array = inf*ones(1,length(actions_array));
    for j=1:length(actions_array)
        theta_posterior_tmp = updateTheta(actions_array(j), expected_depths_by_actions(:,j), theta_prior1, forward_NN);
        %disp(length(theta_posterior_tmp))
        entropy_array(j) = entropy_calculation(theta_posterior_tmp);
    end
    
    indexes = find(entropy_array == min(entropy_array));
    index = indexes(1);
    min_entropy = entropy_array(index)
    action = actions_array(index)
    d0 = expected_depths_by_actions(:,index);
    theta_posterior = updateTheta(action, d0, theta_prior1, forward_NN);
    theta_prior1 = theta_posterior;
    %expected_depths_by_actions = compute_expected_depths_by_actions(theta_prior1, actions_array, forward_NN);
end

%%
% plots
figure;
subplot(3,1,1);
hist(theta_posterior(:,1),yield_low+0.05:0.1:yield_high-0.05);
%xlim([yield_low yield_high])
ylabel('\sigma_{yield}')
subplot(3,1,2);
hist(theta_posterior(:,2),E1_low+0.1:0.2:E1_high-0.1);
%xlim([E1_low E1_high])
ylabel('E_e')
subplot(3,1,3);
hist(theta_posterior(:,3),E2_low+0.1:0.2:E2_high-0.1);
%xlim([E2_low E2_high])
ylabel('E_{Tiso}')


%%
function entropy = entropy_calculation(posterior)
    global E1_low E1_high E2_low E2_high yield_low yield_high;
    [counts1,centers1] = hist(posterior(:,2), E1_low+0.1:0.2:E1_high-0.1);
    prob1 = counts1./length(posterior);
    prob1 = prob1(prob1~=0);
    entropy1 = -sum(prob1.*log(prob1));
    [counts2,centers2] = hist(posterior(:,3), E2_low+0.1:0.2:E2_high-0.1);
    prob2 = counts2./length(posterior);
    prob2 = prob2(prob2~=0);
    entropy2 = -sum(prob2.*log(prob2));
    [counts3,centers3] = hist(posterior(:,1), yield_low+0.05:0.1:yield_high-0.05);
    prob3 = counts3./length(posterior);
    prob3 = prob3(prob3~=0);
    entropy3 = -sum(prob3.*log(prob3));
    entropy = mean([entropy1; entropy2; entropy3]);
end

function theta_posterior = updateTheta(action, response, theta_prior,forward_NN)
    abc_threshold = 0.005;
    theta_posterior = ABC_Method(response',abc_threshold,action,theta_prior,forward_NN);
end

function Posterior_distribution = ABC_Method(observed_data, threshold, action, prior, forward_NN)
    distance_function = @(x,y) mean(abs(x-y),2);
    Posterior_distribution = [];
    ys = prior(:,1);
    E1 = prior(:,2);
    E2 = prior(:,3);
    % generate the sim data 
    X = predict(forward_NN, [action*ones(length(prior),1),ys,E1,E2]);
    % calcalute the distance from Y 
    distance = distance_function(X,observed_data);
    %disp('X:');disp(X);
    %disp('observed:');disp(observed_data);
    %disp(distance);
    Posterior_distribution = prior(distance < threshold,:);
end

function expected_depths = compute_expected_depths_by_actions(theta_prior, actions_array, forward_NN)
    expected_depths = inf*ones(2,length(actions_array));    
    for i = 1:length(actions_array)
        action = actions_array(i);
        depths = predict(forward_NN, [action*ones(length(theta_prior),1), theta_prior]);
        expected_depth = mean(depths,1);
        expected_depths(:,i) = expected_depth';
    end
end

function depths = compute_depths_by_actions(theta, actions_array, forward_NN)
    depths = inf*ones(2,length(actions_array));
    for i = 1:length(actions_array)
        action = actions_array(i);
        depth = predict(forward_NN, [action, theta]);
        depths(:,i) = depth';
    end
end