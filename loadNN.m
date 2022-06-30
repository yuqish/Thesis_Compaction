clc;
clear;

%% import trained model from Python
%M = readtable('elasto-plastic-pivot.csv');
modelfile = 'model1.json';
weights = 'model1.h5';
forward_NN = importKerasNetwork(modelfile,'WeightFile',weights,'OutputLayerType','regression');
%dlnet = dlnetwork(layers);
%YPredicted = predict(net,XValidation);

modelfile = 'model2.json';
weights = 'model2.h5';
backward_NN = importKerasNetwork(modelfile,'WeightFile',weights,'OutputLayerType','regression');

%%  online learning
import com.comsol.model.util.*
comsol_model = mphopen('elastoplastic');
%comsol_model.study('std2').run;
%V = mphevalpoint(comsol_model,'v','selection',8,'dataset','dset9','t',[1.5,3])
%comsol_model.param.set('asphalt_E1',4e6);


% compaction pressure bounds
l_low = 1.6;
l_high = 2;
%other params bounds
global E1_low E1_high E2_low E2_high yield_low yield_high;
E1_low = 2.5;
E1_high = 7.5;
E2_low = 0.5;
E2_high = 6.5;
yield_low = 0.1;
yield_high = 1.3;

% assume a uniform prior on parameters, generate prior
sample_size = 10000;
randm = rand(sample_size,3);
randm(:,1) = randm(:,1)*(yield_high-yield_low) + yield_low;
randm(:,2) = randm(:,2)*(E1_high-E1_low) + E1_low;
randm(:,3) = randm(:,3)*(E2_high-E2_low) + E2_low;
theta_prior = [];
for i = 1:sample_size
    if randm(i,2) > randm(i,3)
        theta_prior = [theta_prior;randm(i,:)];
    end
end
disp(length(theta_prior))

figure;
subplot(3,1,1);
hist(theta_prior(:,1));
xlim([yield_low yield_high])
ylabel('\sigma_{yield}')
subplot(3,1,2);
hist(theta_prior(:,2));
xlim([E1_low E1_high])
ylabel('E_e')
subplot(3,1,3);
hist(theta_prior(:,3));
xlim([E2_low E2_high])
ylabel('E_{Tiso}')
%%
% actual
%global yield; global E1; global E2;
%yield = 0.7;
%E1 = 4;
%E2 = 1;
yield = 0.3;
E1 = 6;
E2 = 3;
%yield = 0.5;
%E1 = 5;
%E2 = 2;
comsol_model.param.set('asphalt_yield',yield*10^6);
comsol_model.param.set('asphalt_E1',E1*10^6);
comsol_model.param.set('asphalt_dE',(E1-E2)*10^6);

% target compaction depth
%d_target = -0.05;
d_target = -0.03;
%epsilon-greedy
%epsilon = 0.5;
n_trials = 30;
%actions_array = [1.6, 1.65, 1.7, 1.75, 1.8, 1.85, 1.9, 1.95, 2];
actions_array = 1.6:0.02:2;

global explored_actions;

epsilon_array = 0:0.1:1;
response_array = zeros(11,n_trials);
entropy_array = zeros(11,n_trials);
for jj = 1:length(epsilon_array)
%for jj = 2
    epsilon = epsilon_array(jj);
    expected_depths_by_actions = inf*ones(2,length(actions_array));
    depths_by_actions = zeros(2,length(actions_array));
    explored_actions = [];
    theta_prior1 = theta_prior;
    for ii = 1:n_trials
        expected_depths_by_actions = update_expected_depths_by_actions(theta_prior1, actions_array, expected_depths_by_actions, forward_NN);
        action = eGreedy(ii, n_trials, epsilon, theta_prior1, actions_array, expected_depths_by_actions, depths_by_actions, forward_NN, d_target);
        response = comsol_response(comsol_model, action);
        fprintf('response: %4.4f\n',response)
        response_array(jj,ii) = response(2);
        theta_posterior = updateTheta(action, response, theta_prior1, forward_NN);
        entropy_array(jj,ii) = entropy_calculation(theta_posterior);
        theta_prior1 = theta_posterior;
        depths_by_actions = update_depths_by_actions(actions_array, action, response, depths_by_actions);
    end

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
end

%
depthmse = zeros(1,size(response_array,1));
variance = zeros(1,size(response_array,1));
for i = 1:size(response_array,1)
    depthmse(i) = sum(abs(response_array(i,:)-d_target));
    variance(i) = sum(entropy_array(i,:));
    fprintf('total reward: %4.8f\n',depthmse(i))
end
%
figure;
hold on
for ii = 1:length(depthmse)
    x = depthmse(ii);
    y = variance(ii);
    scatter(x,y,'filled','DisplayName',['\epsilon=',num2str(epsilon_array(ii))])
end
legend
xlabel('sum of absolute difference from target depth')
ylabel('sum of \theta posterior entropy')
hold off
%
figure;
subplot(1,2,1)
plot(0:0.1:1, depthmse, 'o')
xlabel('\epsilon')
ylabel('sum of absolute difference from target depth')
subplot(1,2,2)
plot(0:0.1:1, variance, 'o')
xlabel('\epsilon')
ylabel('sum of \theta posterior entropy')

%% step function
figure;
hold on
%stairs(1:50, 0.03*ones(1,50))
X = linspace(1,n_trials,n_trials)';
Y = [-0.03*ones(n_trials,1), response_array(5,:)'];
h = stairs(X,Y);
ylabel('depth change (m)')
xlim([1,n_trials]);
ylim([-0.06, 0]);
legend('target depth change', 'compaction history')
hold off
%%
% compute ABC
function Posterior_distribution = ABC_Method_old(observed_data, threshold, action, prior, forward_NN)
    distance_function = @(x,y) mean(abs(x-y),2);
    Posterior_distribution = [];
    for i = 1:length(prior)
        ys = prior(i,1);
        E1 = prior(i,2);
        E2 = prior(i,3);
        % generate the sim data 
        X = predict(forward_NN, [action,ys,E1,E2]);
        % calcalute the distance from Y 
        distance = distance_function(X,observed_data);
        %disp('X:');disp(X);
        %disp('observed:');disp(observed_data);
        %disp(distance);
        if distance < threshold
            fprintf('ys: %4.4f, E1: %4.4f, E2: %4.4f\n',ys,E1,E2);
            Posterior_distribution = [Posterior_distribution; ys,E1,E2];
        end
    end
end

function Posterior_distribution = ABC_Method(observed_data, threshold, action, prior, forward_NN)
    distance_function = @(x,y) mean(abs(x-y),2);
    Posterior_distribution = [];
    ys = prior(:,1);
    E1 = prior(:,2);
    E2 = prior(:,3);
    % generate the sim data 
    X = predict(forward_NN, [action*ones(length(prior),1),ys,E1,E2]);
    %disp(X)
    disp(observed_data)
    % calcalute the distance from Y 
    distance = distance_function(X,observed_data);
    %disp('X:');disp(X);
    %disp('observed:');disp(observed_data);
    %disp(distance);
    Posterior_distribution = prior(distance < threshold,:);
end


function response = comsol_response(comsol_model, action)
    %global yield; global E1; global E2;
    comsol_model.param.set('pressure_compact',round(action,2)*10^6);
    comsol_model.study('std2').run;
    response = mphevalpoint(comsol_model,'v','selection',8,'dataset','dset9','t',[1.5,3]);
end

function action = eGreedy_old(iter, n, epsilon,theta_prior, l_high, l_low, backward_NN, d_target)
    if rand < epsilon
    %if iter < n*epsilon
        action = rand*(l_high-l_low) + l_low;
        action = round(action,2);
        fprintf('random action: %4.4f\n',action);
    else
        input_vector = [theta_prior, d_target*ones(length(theta_prior),1)];
        actions = predict(backward_NN,input_vector);
        % filter by range
        actions = actions(actions<=2 & actions>=1.6);
        action = mean(actions,1);
        action = round(action,2);
        fprintf('best action: %4.4f\n',action);
    end
end

function action = eGreedy(iter, n, epsilon, theta_prior, actions, expected_depths, observed_depths, forward_NN, d_target)
    global explored_actions;
    %if rand < epsilon
    if iter <= n*epsilon
    %if iter > (1-epsilon)*n
        posterior_variance = inf*ones(1,length(actions));

        for i = 1:length(actions)
            %explored = 0;
            %for j = 1:length(explored_actions)
            %    if actions(i) == explored_actions(j)
            %        explored = 1;
            %    end
            %end
            %if explored == 1
            %    continue;
            %end
            action_tmp = actions(i);
            if observed_depths(2,i) == 0  % not observed before
                depth = expected_depths(:,i);
            else
                depth = observed_depths(:,i);
            end
            threshold = 0.005;
            posterior = ABC_Method(depth', threshold, action_tmp, theta_prior, forward_NN);
            entropy = entropy_calculation(posterior);
            fprintf('i=%d, posterior entropy: %4.4f\n',i,entropy);
            posterior_variance(i) = entropy;
        end
        action = actions(posterior_variance==min(posterior_variance));
        action = action(1); % in case there are two optimal actions
        fprintf('random action: %4.4f\n',action);
        explored_actions = [explored_actions action];
    else
        abs_target = abs(expected_depths(2,:)-d_target);
        action = actions(abs_target == min(abs_target));
        action = action(1); % in case there are two optimal actions
        fprintf('best action: %4.4f\n',action);
    end
end

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
    theta_posterior = ABC_Method(transpose(response),abc_threshold,action,theta_prior,forward_NN);
end

function expected_depths = update_expected_depths_by_actions(theta_prior, actions_array, expected_depths_old, forward_NN)
    for i = 1:length(actions_array)
        action = actions_array(i);
        depths = predict(forward_NN, [action*ones(length(theta_prior),1), theta_prior]);
        expected_depth = mean(depths,1);
        expected_depths_old(:,i) = expected_depth';
    end
    expected_depths = expected_depths_old;
end

%explored depths
function depths = update_depths_by_actions(actions_array, action, response, depths_old)
    index = action == actions_array;
    if depths_old(2,index) == 0
        depths_old(:,index) = response;
    else
        % mean depth among all observations
        depths_old(:,index) = (depths_old(:,index) + response)/2;
    end
    depths = depths_old;
end
