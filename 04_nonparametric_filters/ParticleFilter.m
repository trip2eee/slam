classdef ParticleFilter
  properties
    M = 0;        % The number of particls
    px = 0;       % Probability
    w = [];       % weight
    X = [];       % samples
  endproperties
  methods
    function obj = ParticleFilter(M, px)
      obj.M = M;
      obj.px = px;
      obj.w = zeros(M,1);
      obj.X = [];
    endfunction
  endmethods
end




