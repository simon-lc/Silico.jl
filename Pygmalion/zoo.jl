################################################################################
# recipient
################################################################################
A0_recipient = [
	+1.0 +0.0;
	+0.0 +1.0;
	-1.0 +0.0;
	+0.0 -1.0;
	]
normalize_A!(A0_recipient)
b0_recipient = [
	+0.30,
	+0.10,
	+0.30,
	+0.10,
	]
o0_recipient = [0.0, -0.10]

A1_recipient = [
	+1.0 +0.0;
	+0.0 +1.0;
	-1.0 +0.0;
	+0.0 -1.0;
	]
normalize_A!(A1_recipient)
b1_recipient = [
	+0.10,
	+0.25,
	+0.10,
	+0.25,
	]
o1_recipient = [+0.4, 0.05]

A2_recipient = [
	+1.0 +0.0;
	+0.0 +1.0;
	-1.0 +0.0;
	+0.0 -1.0;
	]
normalize_A!(A2_recipient)
b2_recipient = [
	+0.10,
	+0.25,
	+0.10,
	+0.25,
	]
o2_recipient = [-0.4, 0.05]

A_recipient = [A0_recipient, A1_recipient, A2_recipient]
b_recipient = [b0_recipient, b1_recipient, b2_recipient]
o_recipient = [o0_recipient, o1_recipient, o2_recipient]
bo_recipient = [
	b0_recipient + A0_recipient * o0_recipient,
	b1_recipient + A1_recipient * o1_recipient,
	b2_recipient + A2_recipient * o2_recipient,
	]



################################################################################
# recipient
################################################################################
A0_box = [
	+1.0 +0.0;
	+0.0 +1.0;
	-1.0 +0.0;
	+0.0 -1.0;
	]
normalize_A!(A0_box)
b0_box = [
	+0.50,
	+0.50,
	+0.50,
	+0.50,
	]
o0_box = [0.0, 0.0]

A_box = [A0_box,]
b_box = [b0_box,]
o_box = [o0_box,]
bo_box = [
	b0_box + A0_box * o0_box,
	]
