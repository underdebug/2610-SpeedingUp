#pragma once

#define DEFAULT_BLOCK_SIZE	256

class AppConfig {
private:
	const int profile_substeps = 1;
	const int sim_substeps = 30;
	int block_size;
	bool profile_mode;
	int substeps;

public:
	AppConfig(int block_size, bool profile_mode = false) {
		this->block_size = block_size;
		this->profile_mode = profile_mode;
		substeps = profile_mode ? profile_substeps : sim_substeps;
	}

	const int profile_loop_count = 10;
	int get_block_size() const { return block_size; }
	int get_substeps() const { return substeps; }
	bool is_profile_mode() const { return profile_mode; }
};