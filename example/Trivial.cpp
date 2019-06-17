#include "../include/OCLRadixSort.hpp"

int main() {
  // create instance for device
  bc::device device = bc::system::default_device();
  bc::context context = bc::context(device);
  bc::command_queue command_queue = bc::command_queue(context, device);

  RadixSort<> rs(context, command_queue);

  std::vector<unsigned> keys_host(1 << 25);
  std::cout << "generating " << keys_host.size() << " keys_host...." << std::endl;
  std::generate(keys_host.begin(), keys_host.end(), rand);

  bc::vector<unsigned> keys(keys_host.begin(), keys_host.end(), command_queue);
  bc::vector<double> values(keys_host.size(), context);

  rs.sort(keys, values);

  bc::copy(keys.begin(), keys.end(), keys_host.begin(), command_queue);

  std::cout << "Sorted: " << std::boolalpha
            << std::is_sorted(keys_host.begin(), keys_host.end()) << std::endl;
}
